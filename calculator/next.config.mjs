/** @type {import('next').NextConfig} */
const nextConfig = {
  // The calculator is a local tool. It talks to one route handler, which shells
  // out to the Python package in the repository root.
  reactStrictMode: true,
};

export default nextConfig;
